"""
Cochlear filterbank: mechanical stage of the model.

Implements:
  - Greenwood (1990) function for the tonotopic map (position -> characteristic frequency)
  - ERB (Glasberg & Moore, 1990) scale for realistic auditory filter bandwidths
  - Gammatone filters (Slaney 1993 implementation) as the standard proxy for
    basilar membrane traveling-wave filtering at each cochlear place.
"""

import numpy as np
from scipy.signal import lfilter


def greenwood_freq(x, A=165.4, a=2.1, k=0.88):
    """
    Greenwood (1990) map: cochlear position (0=apex,1=base) -> characteristic frequency (Hz).
    Default constants are the standard human cochlea fit.
    x: fractional distance along the basilar membrane, 0 (apex/low-freq) to 1 (base/high-freq)
    """
    return A * (10 ** (a * x) - k)


def greenwood_position(f, A=165.4, a=2.1, k=0.88):
    """Inverse Greenwood map: frequency (Hz) -> cochlear position (0..1)."""
    return np.log10(f / A + k) / a


def erb_bandwidth(fc):
    """
    Glasberg & Moore (1990) Equivalent Rectangular Bandwidth.
    fc: center frequency in Hz. Returns ERB in Hz.
    Auditory filters get proportionally narrower (in relative terms) but
    absolutely wider at higher center frequencies -- this is the realistic
    nonuniform bandwidth that distinguishes a real cochlear model from a
    naive linearly-spaced bandpass bank.
    """
    return 24.7 * (4.37 * fc / 1000.0 + 1.0)


def make_erb_spaced_freqs(low_freq, high_freq, n_channels):
    """
    Space n_channels characteristic frequencies evenly along the Greenwood
    (cochlear-position) map between low_freq and high_freq. This reproduces
    the real cochlea's property that channel density is uniform in cochlear
    *position*, not in Hz -- hence the channels bunch up at low frequencies.
    """
    x_low = greenwood_position(low_freq)
    x_high = greenwood_position(high_freq)
    positions = np.linspace(x_low, x_high, n_channels)
    cfs = greenwood_freq(positions)
    return cfs, positions


class GammatoneFilterbank:
    """
    Bank of gammatone filters, one per cochlear channel, implemented as
    cascaded 4th-order IIR filters (Slaney 1993 'Auditory Toolbox' design).
    This is the standard signal-processing stand-in for basilar-membrane
    traveling-wave mechanics: each filter's center frequency and bandwidth
    come directly from the Greenwood/ERB biology above, not arbitrary choices.
    """

    def __init__(self, fs, cfs):
        self.fs = fs
        self.cfs = np.asarray(cfs)
        self.n_channels = len(cfs)
        self._design_filters()

    def _design_filters(self):
        fs = self.fs
        T = 1.0 / fs
        self.coeffs = []
        for fc in self.cfs:
            b_erb = erb_bandwidth(fc)
            # Slaney's gammatone coefficient derivation
            order = 1
            bw = 1.019 * 2 * np.pi * b_erb
            fc_rad = 2 * np.pi * fc

            A0 = T
            A2 = 0.0
            B0 = 1.0
            B1 = -2 * np.cos(fc_rad * T) * np.exp(-bw * T)
            B2 = np.exp(-2 * bw * T)

            A11 = -(2 * T * np.cos(fc_rad * T) / np.exp(bw * T) +
                    2 * np.sqrt(3 + 2 ** 1.5) * T * np.sin(fc_rad * T) / np.exp(bw * T)) / 2
            A12 = -(2 * T * np.cos(fc_rad * T) / np.exp(bw * T) -
                    2 * np.sqrt(3 + 2 ** 1.5) * T * np.sin(fc_rad * T) / np.exp(bw * T)) / 2
            A13 = -(2 * T * np.cos(fc_rad * T) / np.exp(bw * T) +
                    2 * np.sqrt(3 - 2 ** 1.5) * T * np.sin(fc_rad * T) / np.exp(bw * T)) / 2
            A14 = -(2 * T * np.cos(fc_rad * T) / np.exp(bw * T) -
                    2 * np.sqrt(3 - 2 ** 1.5) * T * np.sin(fc_rad * T) / np.exp(bw * T)) / 2

            gain = np.abs(
                (-2 * np.exp(4j * fc_rad * T) * T +
                 2 * np.exp(-(bw * T) + 2j * fc_rad * T) * T *
                 (np.cos(fc_rad * T) - np.sqrt(3 - 2 ** 1.5) * np.sin(fc_rad * T))) *
                (-2 * np.exp(4j * fc_rad * T) * T +
                 2 * np.exp(-(bw * T) + 2j * fc_rad * T) * T *
                 (np.cos(fc_rad * T) + np.sqrt(3 - 2 ** 1.5) * np.sin(fc_rad * T))) *
                (-2 * np.exp(4j * fc_rad * T) * T +
                 2 * np.exp(-(bw * T) + 2j * fc_rad * T) * T *
                 (np.cos(fc_rad * T) - np.sqrt(3 + 2 ** 1.5) * np.sin(fc_rad * T))) *
                (-2 * np.exp(4j * fc_rad * T) * T +
                 2 * np.exp(-(bw * T) + 2j * fc_rad * T) * T *
                 (np.cos(fc_rad * T) + np.sqrt(3 + 2 ** 1.5) * np.sin(fc_rad * T))) /
                (-2 / np.exp(2 * bw * T) - 2 * np.exp(4j * fc_rad * T) +
                 2 * (1 + np.exp(4j * fc_rad * T)) / np.exp(bw * T)) ** 4
            )

            stage_b = [[A0 / gain, A11 / gain, A2 / gain],
                       [A0, A12, A2],
                       [A0, A13, A2],
                       [A0, A14, A2]]
            stage_a = [B0, B1, B2]
            self.coeffs.append((stage_b, stage_a))

    def process(self, x):
        """
        Run x (1D audio array) through all channels.
        Returns array of shape (n_channels, len(x)).
        """
        out = np.zeros((self.n_channels, len(x)))
        for i, (stage_b, stage_a) in enumerate(self.coeffs):
            y = x.copy()
            for b in stage_b:
                y = lfilter(b, stage_a, y)
            out[i] = y
        return out


if __name__ == "__main__":
    # quick sanity check: pure tones should peak in the channel nearest their CF
    fs = 20000
    cfs, positions = make_erb_spaced_freqs(100, 8000, 16)
    print("Channel CFs (Hz):", np.round(cfs, 1))

    fb = GammatoneFilterbank(fs, cfs)

    t = np.arange(0, 0.2, 1 / fs)
    for test_freq in [200, 1000, 4000]:
        tone = np.sin(2 * np.pi * test_freq * t)
        resp = fb.process(tone)
        energy = np.sqrt(np.mean(resp ** 2, axis=1))
        peak_ch = np.argmax(energy)
        print(f"{test_freq} Hz tone -> peak channel {peak_ch} (CF={cfs[peak_ch]:.0f} Hz)")
