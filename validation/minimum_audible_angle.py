import numpy as np
import matplotlib.pyplot as plt


def get_noise(T, amplitude=1, freq_range=[10000, 30000], sampling_rate=100000):
    length = int(T * sampling_rate)
    a = np.random.randn(length//2+1) # faster than white = np.random.randn(length), a = np.fft.rfft(white)
    nyquist_freq = sampling_rate / 2
    nyquist_index = length//2
    low_index = int(freq_range[0] / nyquist_freq * nyquist_index)
    high_index = int(freq_range[1] / nyquist_freq * nyquist_index)
    a[:low_index] = 0
    a[high_index:] = 0
    result = np.fft.irfft(a)
    rms = np.sqrt(np.mean(result**2))
    result = result / rms * amplitude
    return result


class MockModel:
    def __init__(self):
        pass

    def get_angle_estimate(self, stimulus, true_angle):
        """
        :param stimulus: monophonic signal
        :param true_angle: angle at which stimulus is presented
        :return: sound source angle estimated by model
        """
        return true_angle + np.random.randn() * 30


def get_threshold(model):
    """
    :param model: finds decision threshold that matches 4% false alarm rate in Behrens & Klump (2016)
    :return: minimum estimated angle magnitude at which model decides angle is nonzero
    """
    false_alarm_rate = .04

    reference_estimates = []
    for i in range(100):
        stimulus = get_noise(.1)
        estimate = model.get_angle_estimate(stimulus, 0)
        reference_estimates.append(estimate)

    deviations = np.abs(reference_estimates)
    return np.percentile(deviations, 100*(1-false_alarm_rate))


def plot_hit_rates(model):
    """
    :param model: Model wrapper with same interface as MockModel
    """
    threshold = get_threshold(model)

    n = 50
    angles = [0, 13, 26, 39, 51, 64, 77, 90]
    hit_rates = []
    for angle in angles:
        hit_count = 0
        for i in range(n):
            stimulus = get_noise(.1)
            estimate = model.get_angle_estimate(stimulus, angle)
            if np.abs(estimate) > threshold:
                hit_count = hit_count + 1
        hit_rates.append(hit_count / n)

    mouse_hit_rates = [ # Behrens & Klump Fig. 5 via WebPlotDigitizer
        0.034,
        0.038,
        0.155,
        0.391,
        0.451,
        0.609,
        0.698,
        0.699
    ]
    mouse_standard_errors = [
        0.008,
        0.01,
        0.082,
        0.133,
        0.151,
        0.055,
        0.071,
        0.069
    ]

    plt.plot(angles, hit_rates, 'k.-')
    plt.errorbar(angles, mouse_hit_rates, yerr=mouse_standard_errors)
    plt.legend(('Model', 'Mice'))
    plt.xlabel('Angle')
    plt.ylabel('Hit Rate')
    plt.show()


if __name__ == '__main__':
    # plot power spectrum of noise stimulus
    # sampling_rate = 100000
    # noise = get_noise(.1, 1, sampling_rate=sampling_rate)
    # frequencies = np.fft.rfftfreq(len(noise), d=1 / sampling_rate)
    # a = np.fft.rfft(noise)
    # power = np.abs(a) ** 2 / len(noise)
    # plt.plot(frequencies, power)
    # plt.show()

    # plot hit rates for mock model
    plot_hit_rates(MockModel())